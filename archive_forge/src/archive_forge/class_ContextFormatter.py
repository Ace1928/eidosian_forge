import datetime
import debtcollector
import functools
import io
import itertools
import logging
import logging.config
import logging.handlers
import re
import socket
import sys
import traceback
from dateutil import tz
from oslo_context import context as context_utils
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
class ContextFormatter(logging.Formatter):
    """A context.RequestContext aware formatter configured through flags.

    The flags used to set format strings are: logging_context_format_string
    and logging_default_format_string.  You can also specify
    logging_debug_format_suffix to append extra formatting if the log level is
    debug.

    The standard variables available to the formatter are listed at:
    http://docs.python.org/library/logging.html#formatter

    In addition to the standard variables, one custom variable is
    available to both formatting string: `isotime` produces a
    timestamp in ISO8601 format, suitable for producing
    RFC5424-compliant log messages.

    Furthermore, logging_context_format_string has access to all of
    the data in a dict representation of the context.
    """

    def __init__(self, *args, **kwargs):
        """Initialize ContextFormatter instance

        Takes additional keyword arguments which can be used in the message
        format string.

        :keyword project: project name
        :type project: string
        :keyword version: project version
        :type version: string

        """
        self.project = kwargs.pop('project', 'unknown')
        self.version = kwargs.pop('version', 'unknown')
        self.conf = kwargs.pop('config', _CONF)
        logging.Formatter.__init__(self, *args, **kwargs)

    def format(self, record):
        """Uses contextstring if request_id is set, otherwise default."""
        record.project = self.project
        record.version = self.version
        instance_extra = ''
        instance = getattr(record, 'instance', None)
        instance_uuid = getattr(record, 'instance_uuid', None)
        context = _update_record_with_context(record)
        if instance:
            try:
                instance_extra = self.conf.instance_format % instance
            except TypeError:
                instance_extra = instance
        elif instance_uuid:
            instance_extra = self.conf.instance_uuid_format % {'uuid': instance_uuid}
        elif context:
            instance = getattr(context, 'instance', None)
            instance_uuid = getattr(context, 'instance_uuid', None)
            resource_uuid = getattr(context, 'resource_uuid', None)
            if instance:
                instance_extra = self.conf.instance_format % {'uuid': instance}
            elif instance_uuid:
                instance_extra = self.conf.instance_uuid_format % {'uuid': instance_uuid}
            elif resource_uuid:
                instance_extra = self.conf.instance_uuid_format % {'uuid': resource_uuid}
        record.instance = instance_extra
        for key in ('instance', 'color', 'user_identity', 'resource', 'user_name', 'project_name', 'global_request_id'):
            if key not in record.__dict__:
                record.__dict__[key] = ''
        if context:
            record.user_identity = self.conf.logging_user_identity_format % _ReplaceFalseValue(_dictify_context(context))
        if record.__dict__.get('request_id'):
            fmt = self.conf.logging_context_format_string
        else:
            fmt = self.conf.logging_default_format_string
        if record.exc_info:
            record.exc_text = self.formatException(record.exc_info, record)
        record.error_summary = _get_error_summary(record)
        if '%(error_summary)s' in fmt:
            record.error_summary = record.error_summary or '-'
        elif record.error_summary:
            fmt += ': %(error_summary)s'
        if record.levelno == logging.DEBUG and self.conf.logging_debug_format_suffix:
            fmt += ' ' + self.conf.logging_debug_format_suffix
        self._compute_iso_time(record)
        self._style = logging.PercentStyle(fmt)
        self._fmt = self._style._fmt
        try:
            return logging.Formatter.format(self, record)
        except TypeError as err:
            record.msg = 'Error formatting log line msg={!r} err={!r}'.format(record.msg, err).replace('%', '*')
            return logging.Formatter.format(self, record)

    def formatException(self, exc_info, record=None):
        """Format exception output with CONF.logging_exception_prefix."""
        if not record:
            try:
                return logging.Formatter.formatException(self, exc_info)
            except TypeError as type_error:
                msg = str(type_error)
                return '<Unprintable exception due to %s>\n' % msg
        stringbuffer = io.StringIO()
        try:
            traceback.print_exception(exc_info[0], exc_info[1], exc_info[2], None, stringbuffer)
        except TypeError as type_error:
            msg = str(type_error)
            stringbuffer.write('<Unprintable exception due to %s>\n' % msg)
        lines = stringbuffer.getvalue().split('\n')
        stringbuffer.close()
        if self.conf.logging_exception_prefix.find('%(asctime)') != -1:
            record.asctime = self.formatTime(record, self.datefmt)
        self._compute_iso_time(record)
        formatted_lines = []
        for line in lines:
            pl = self.conf.logging_exception_prefix % record.__dict__
            fl = '%s%s' % (pl, line)
            formatted_lines.append(fl)
        return '\n'.join(formatted_lines)

    def _compute_iso_time(self, record):
        localtz = tz.tzlocal()
        record.isotime = datetime.datetime.fromtimestamp(record.created).replace(tzinfo=localtz).isoformat()
        if record.created == int(record.created):
            record.isotime = '%s.000000%s' % (record.isotime[:-6], record.isotime[-6:])
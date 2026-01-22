import logging
import os
import re
import socket
from urllib import parse
import netaddr
from netaddr.core import INET_PTON
import netifaces
from oslo_utils._i18n import _
class _ModifiedSplitResult(parse.SplitResult):
    """Split results class for urlsplit."""

    def params(self, collapse=True):
        """Extracts the query parameters from the split urls components.

        This method will provide back as a dictionary the query parameter
        names and values that were provided in the url.

        :param collapse: Boolean, turn on or off collapsing of query values
        with the same name. Since a url can contain the same query parameter
        name with different values it may or may not be useful for users to
        care that this has happened. This parameter when True uses the
        last value that was given for a given name, while if False it will
        retain all values provided by associating the query parameter name with
        a list of values instead of a single (non-list) value.
        """
        if self.query:
            if collapse:
                return dict(parse.parse_qsl(self.query))
            else:
                params = {}
                for key, value in parse.parse_qsl(self.query):
                    if key in params:
                        if isinstance(params[key], list):
                            params[key].append(value)
                        else:
                            params[key] = [params[key], value]
                    else:
                        params[key] = value
                return params
        else:
            return {}
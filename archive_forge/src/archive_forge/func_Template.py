from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
import collections
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.run import condition
from googlecloudsdk.core.console import console_attr
import six
@classmethod
def Template(cls, template, messages_mod, kind=None):
    """Wraps a template object: spec and metadata only, no status."""
    msg_cls = getattr(messages_mod, cls.Kind(kind))
    return cls(msg_cls(spec=template.spec, metadata=template.metadata), messages_mod, kind)
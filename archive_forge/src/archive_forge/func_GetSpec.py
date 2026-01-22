from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.api_lib.compute import property_selector
import six
import six.moves.http_client
def GetSpec(resource_type, message_classes, api_version):
    """Returns a Spec for the given resource type."""
    spec = _GetSpecsForVersion(api_version)
    if resource_type not in spec:
        raise KeyError('"%s" not found in Specs for version "%s"' % (resource_type, api_version))
    spec = spec[resource_type]
    table_cols = []
    for name, action in spec.table_cols:
        if isinstance(action, six.string_types):
            table_cols.append((name, property_selector.PropertyGetter(action)))
        elif callable(action):
            table_cols.append((name, action))
        else:
            raise ValueError('expected function or property in table_cols list: {0}'.format(spec))
    message_class = getattr(message_classes, spec.message_class_name)
    fields = list(_ProtobufDefinitionToFields(message_class))
    return Spec(message_class=message_class, fields=fields, table_cols=table_cols, transformations=spec.transformations, editables=spec.editables)
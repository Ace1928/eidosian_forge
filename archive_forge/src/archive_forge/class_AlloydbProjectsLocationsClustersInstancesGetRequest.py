from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AlloydbProjectsLocationsClustersInstancesGetRequest(_messages.Message):
    """A AlloydbProjectsLocationsClustersInstancesGetRequest object.

  Enums:
    ViewValueValuesEnum: The view of the instance to return.

  Fields:
    name: Required. The name of the resource. For the required format, see the
      comment on the Instance.name field.
    view: The view of the instance to return.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """The view of the instance to return.

    Values:
      INSTANCE_VIEW_UNSPECIFIED: INSTANCE_VIEW_UNSPECIFIED Not specified,
        equivalent to BASIC.
      INSTANCE_VIEW_BASIC: BASIC server responses for a primary or read
        instance include all the relevant instance details, excluding the
        details of each node in the instance. The default value.
      INSTANCE_VIEW_FULL: FULL response is equivalent to BASIC for primary
        instance (for now). For read pool instance, this includes details of
        each node in the pool.
    """
        INSTANCE_VIEW_UNSPECIFIED = 0
        INSTANCE_VIEW_BASIC = 1
        INSTANCE_VIEW_FULL = 2
    name = _messages.StringField(1, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 2)
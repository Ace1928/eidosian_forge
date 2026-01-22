from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConsentAccessorScope(_messages.Message):
    """The accessor scope that describes who can access, for what purpose, in
  which environment.

  Fields:
    actor: An individual, group, or access role that identifies the accessor
      or a characteristic of the accessor. This can be a resource ID (such as
      `{resourceType}/{id}`) or an external URI. This value must be present.
    environment: An abstract identifier that describes the environment or
      conditions under which the accessor is acting. Can be "*" if it applies
      to all environments.
    purpose: The intent of data use. Can be "*" if it applies to all purposes.
  """
    actor = _messages.StringField(1)
    environment = _messages.StringField(2)
    purpose = _messages.StringField(3)
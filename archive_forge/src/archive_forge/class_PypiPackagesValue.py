from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class PypiPackagesValue(_messages.Message):
    """Optional. Custom Python Package Index (PyPI) packages to be installed
    in the environment. Keys refer to the lowercase package name such as
    "numpy" and values are the lowercase extras and version specifier such as
    "==1.12.0", "[devel,gcp_api]", or "[devel]>=1.8.2, <1.9.2". To specify a
    package without pinning it to a version specifier, use the empty string as
    the value.

    Messages:
      AdditionalProperty: An additional property for a PypiPackagesValue
        object.

    Fields:
      additionalProperties: Additional properties of type PypiPackagesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a PypiPackagesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
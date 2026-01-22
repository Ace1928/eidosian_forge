from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecretKeySelector(_messages.Message):
    """SecretKeySelector selects a key of a Secret.

  Fields:
    key: Required. A Cloud Secret Manager secret version. Must be 'latest' for
      the latest version, an integer for a specific version, or a version
      alias. The key of the secret to select from. Must be a valid secret key.
    localObjectReference: This field should not be used directly as it is
      meant to be inlined directly into the message. Use the "name" field
      instead.
    name: The name of the secret in Cloud Secret Manager. By default, the
      secret is assumed to be in the same project. If the secret is in another
      project, you must define an alias. An alias definition has the form:
      :projects//secrets/. If multiple alias definitions are needed, they must
      be separated by commas. The alias definitions must be set on the
      run.googleapis.com/secrets annotation. The name of the secret in the
      pod's namespace to select from.
    optional: Specify whether the Secret or its key must be defined.
  """
    key = _messages.StringField(1)
    localObjectReference = _messages.MessageField('LocalObjectReference', 2)
    name = _messages.StringField(3)
    optional = _messages.BooleanField(4)
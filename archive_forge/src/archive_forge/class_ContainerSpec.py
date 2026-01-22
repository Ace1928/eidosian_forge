from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContainerSpec(_messages.Message):
    """Container Spec.

  Fields:
    defaultEnvironment: Default runtime environment for the job.
    image: Name of the docker container image. E.g., gcr.io/project/some-image
    imageRepositoryCertPath: Cloud Storage path to self-signed certificate of
      private registry.
    imageRepositoryPasswordSecretId: Secret Manager secret id for password to
      authenticate to private registry.
    imageRepositoryUsernameSecretId: Secret Manager secret id for username to
      authenticate to private registry.
    metadata: Metadata describing a template including description and
      validation rules.
    sdkInfo: Required. SDK info of the Flex Template.
  """
    defaultEnvironment = _messages.MessageField('FlexTemplateRuntimeEnvironment', 1)
    image = _messages.StringField(2)
    imageRepositoryCertPath = _messages.StringField(3)
    imageRepositoryPasswordSecretId = _messages.StringField(4)
    imageRepositoryUsernameSecretId = _messages.StringField(5)
    metadata = _messages.MessageField('TemplateMetadata', 6)
    sdkInfo = _messages.MessageField('SDKInfo', 7)
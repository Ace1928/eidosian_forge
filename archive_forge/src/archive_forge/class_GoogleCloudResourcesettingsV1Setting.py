from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class GoogleCloudResourcesettingsV1Setting(_messages.Message):
    """The schema for settings.

  Fields:
    effectiveValue: Output only. The effective value of the setting at the
      given parent resource, evaluated based on the resource hierarchy The
      effective value evaluates to one of the following options, in this
      order. If an option is not valid or doesn't exist, then the next option
      is used: 1. The local setting value on the given resource:
      Setting.local_value 2. If one of the given resource's ancestors in the
      resource hierarchy have a local setting value, the local value at the
      nearest such ancestor. 3. The setting's default value:
      SettingMetadata.default_value 4. An empty value, defined as a `Value`
      with all fields unset. The data type of Value must always be consistent
      with the data type defined in Setting.metadata.
    etag: A fingerprint used for optimistic concurrency. See UpdateSetting for
      more details.
    localValue: The configured value of the setting at the given parent
      resource, ignoring the resource hierarchy. The data type of Value must
      always be consistent with the data type defined in Setting.metadata.
    metadata: Output only. Metadata about a setting which is not editable by
      the end user.
    name: The resource name of the setting. Must be in one of the following
      forms: * `projects/{project_number}/settings/{setting_name}` *
      `folders/{folder_id}/settings/{setting_name}` *
      `organizations/{organization_id}/settings/{setting_name}` For example,
      "/projects/123/settings/gcp-enableMyFeature"
  """
    effectiveValue = _messages.MessageField('GoogleCloudResourcesettingsV1Value', 1)
    etag = _messages.StringField(2)
    localValue = _messages.MessageField('GoogleCloudResourcesettingsV1Value', 3)
    metadata = _messages.MessageField('GoogleCloudResourcesettingsV1SettingMetadata', 4)
    name = _messages.StringField(5)
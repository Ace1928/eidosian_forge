from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigquerydatatransferProjectsTransferConfigsCreateRequest(_messages.Message):
    """A BigquerydatatransferProjectsTransferConfigsCreateRequest object.

  Fields:
    authorizationCode: Optional OAuth2 authorization code to use with this
      transfer configuration. This is required only if
      `transferConfig.dataSourceId` is 'youtube_channel' and new credentials
      are needed, as indicated by `CheckValidCreds`. In order to obtain
      authorization_code, make a request to the following URL: https://www.gst
      atic.com/bigquerydatatransfer/oauthz/auth?redirect_uri=urn:ietf:wg:oauth
      :2.0:oob&response_type=authorization_code&client_id=client_id&scope=data
      _source_scopes * The client_id is the OAuth client_id of the a data
      source as returned by ListDataSources method. * data_source_scopes are
      the scopes returned by ListDataSources method. Note that this should not
      be set when `service_account_name` is used to create the transfer
      config.
    parent: Required. The BigQuery project id where the transfer configuration
      should be created. Must be in the format
      projects/{project_id}/locations/{location_id} or projects/{project_id}.
      If specified location and location of the destination bigquery dataset
      do not match - the request will fail.
    serviceAccountName: Optional service account email. If this field is set,
      the transfer config will be created with this service account's
      credentials. It requires that the requesting user calling this API has
      permissions to act as this service account. Note that not all data
      sources support service account credentials when creating a transfer
      config. For the latest list of data sources, read about [using service
      accounts](https://cloud.google.com/bigquery-transfer/docs/use-service-
      accounts).
    transferConfig: A TransferConfig resource to be passed as the request
      body.
    versionInfo: Optional version info. This is required only if
      `transferConfig.dataSourceId` is not 'youtube_channel' and new
      credentials are needed, as indicated by `CheckValidCreds`. In order to
      obtain version info, make a request to the following URL: https://www.gs
      tatic.com/bigquerydatatransfer/oauthz/auth?redirect_uri=urn:ietf:wg:oaut
      h:2.0:oob&response_type=version_info&client_id=client_id&scope=data_sour
      ce_scopes * The client_id is the OAuth client_id of the a data source as
      returned by ListDataSources method. * data_source_scopes are the scopes
      returned by ListDataSources method. Note that this should not be set
      when `service_account_name` is used to create the transfer config.
  """
    authorizationCode = _messages.StringField(1)
    parent = _messages.StringField(2, required=True)
    serviceAccountName = _messages.StringField(3)
    transferConfig = _messages.MessageField('TransferConfig', 4)
    versionInfo = _messages.StringField(5)
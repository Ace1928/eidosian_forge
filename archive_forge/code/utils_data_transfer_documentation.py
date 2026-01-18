from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import flags
from clients import bigquery_client_extended
from frontend import utils as frontend_utils
Retrieves the authorization code.

  An authorization code is needed if the Data Transfer Service does not
  have credentials for the requesting user and data source. The Data
  Transfer Service will convert this authorization code into a refresh
  token to perform transfer runs on the user's behalf.

  Args:
    reference: The project reference.
    data_source: The data source of the transfer config.
    transfer_client: The transfer api client.

  Returns:
    auth_info: A dict which contains authorization info from user. It is either
    an authorization_code or a version_info.
  
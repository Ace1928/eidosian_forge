import platform
from google.auth import version
def token_request_access_token_mds():
    return '{} {} {}'.format(python_and_auth_lib_version(), REQUEST_TYPE_ACCESS_TOKEN, CRED_TYPE_SA_MDS)
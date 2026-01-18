import platform
from google.auth import version
def token_request_id_token_sa_assertion():
    return '{} {} {}'.format(python_and_auth_lib_version(), REQUEST_TYPE_ID_TOKEN, CRED_TYPE_SA_ASSERTION)
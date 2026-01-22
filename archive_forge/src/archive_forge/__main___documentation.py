import json
import os
import os.path
import click
import google_auth_oauthlib.flow
Command-line tool for obtaining authorization and credentials from a user.

    This tool uses the OAuth 2.0 Authorization Code grant as described
    in section 1.3.1 of RFC6749:
    https://tools.ietf.org/html/rfc6749#section-1.3.1

    This tool is intended for assist developers in obtaining credentials
    for testing applications or samples.

    This is not intended for production use where a combination of
    companion and on-device applications should complete the OAuth 2.0
    authorization flow to get authorization from the users.

    
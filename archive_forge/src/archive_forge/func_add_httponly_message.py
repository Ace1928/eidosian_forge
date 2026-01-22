from django.conf import settings
from .. import Tags, Warning, register
def add_httponly_message(message):
    return message + ' Using an HttpOnly session cookie makes it more difficult for cross-site scripting attacks to hijack user sessions.'
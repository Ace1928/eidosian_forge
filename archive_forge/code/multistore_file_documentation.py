import errno
import json
import logging
import os
import threading
from oauth2client import client
from oauth2client import util
from oauth2client.contrib import locked_file
Delete a credential.

            The Storage lock must be held when this is called.

            Args:
                credentials: Credentials, the credentials to store.
            
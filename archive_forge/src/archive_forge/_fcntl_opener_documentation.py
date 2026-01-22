import errno
import fcntl
import time
from oauth2client.contrib import locked_file
Close and unlock the file using the fcntl.lockf primitive.
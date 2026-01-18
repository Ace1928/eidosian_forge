import grp
import pwd
import socket
from twisted.application import internet, service as appservice
from twisted.internet.protocol import ServerFactory
from twisted.python import log, usage
from twisted.runner import inetd, inetdconf

    To use it, create a file named `sample-inetd.conf` with:

    8123 stream tcp wait some_user /bin/cat -

    You can then run it as in the following example and port 8123 became an
    echo server.

    twistd -n inetd -f sample-inetd.conf
    
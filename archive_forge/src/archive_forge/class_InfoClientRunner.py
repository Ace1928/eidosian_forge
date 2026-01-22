import time
import email
import socket
import logging
import functools
import collections
import pyzor.digest
import pyzor.account
import pyzor.message
import pyzor.hacks.py26
class InfoClientRunner(ClientRunner):

    def handle_response(self, response, message):
        message += '%s\n' % str(response.head_tuple())
        if response.is_ok():
            for key in ('Count', 'Entered', 'Updated', 'WL-Count', 'WL-Entered', 'WL-Updated'):
                if key in response:
                    val = int(response[key])
                    if 'Count' in key:
                        stringed = str(val)
                    elif val == -1:
                        stringed = 'Never'
                    else:
                        stringed = time.ctime(val)
                    message += '\t%s: %s\n' % (key, stringed)
        else:
            self.all_ok = False
        self.results.append(message + '\n')
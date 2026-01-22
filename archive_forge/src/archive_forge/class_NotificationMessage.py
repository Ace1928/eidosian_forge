import hmac
import base64
import re
class NotificationMessage(object):
    NOTIFICATION_WSDL = 'http://mechanicalturk.amazonaws.com/AWSMechanicalTurk/2006-05-05/AWSMechanicalTurkRequesterNotification.wsdl'
    NOTIFICATION_VERSION = '2006-05-05'
    SERVICE_NAME = 'AWSMechanicalTurkRequesterNotification'
    OPERATION_NAME = 'Notify'
    EVENT_PATTERN = 'Event\\.(?P<n>\\d+)\\.(?P<param>\\w+)'
    EVENT_RE = re.compile(EVENT_PATTERN)

    def __init__(self, d):
        """
        Constructor; expects parameter d to be a dict of string parameters from a REST transport notification message
        """
        self.signature = d['Signature']
        self.timestamp = d['Timestamp']
        self.version = d['Version']
        assert d['method'] == NotificationMessage.OPERATION_NAME, "Method should be '%s'" % NotificationMessage.OPERATION_NAME
        self.events = []
        events_dict = {}
        if 'Event' in d:
            events_dict = d['Event']
        else:
            for k in d:
                v = d[k]
                if k.startswith('Event.'):
                    ed = NotificationMessage.EVENT_RE.search(k).groupdict()
                    n = int(ed['n'])
                    param = str(ed['param'])
                    if n not in events_dict:
                        events_dict[n] = {}
                    events_dict[n][param] = v
        for n in events_dict:
            self.events.append(Event(events_dict[n]))

    def verify(self, secret_key):
        """
        Verifies the authenticity of a notification message.

        TODO: This is doing a form of authentication and
              this functionality should really be merged
              with the pluggable authentication mechanism
              at some point.
        """
        verification_input = NotificationMessage.SERVICE_NAME
        verification_input += NotificationMessage.OPERATION_NAME
        verification_input += self.timestamp
        h = hmac.new(key=secret_key, digestmod=sha)
        h.update(verification_input)
        signature_calc = base64.b64encode(h.digest())
        return self.signature == signature_calc
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def uservalue(self):
    if self.uservalue_ is None:
        self.lazy_init_lock_.acquire()
        try:
            if self.uservalue_ is None:
                self.uservalue_ = PropertyValue_UserValue()
        finally:
            self.lazy_init_lock_.release()
    return self.uservalue_
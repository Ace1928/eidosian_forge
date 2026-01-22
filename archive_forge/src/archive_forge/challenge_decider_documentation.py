import re
from paste.httpheaders import CONTENT_TYPE
from paste.httpheaders import REQUEST_METHOD
from paste.httpheaders import USER_AGENT
from paste.request import construct_url
from repoze.who.interfaces import IRequestClassifier
import zope.interface
Returns one of the classifiers 'dav', 'xmlpost', or 'browser',
    depending on the imperative logic below
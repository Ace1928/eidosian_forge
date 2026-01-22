import xml.sax
import datetime
import itertools
from boto import handler
from boto import config
from boto.mturk.price import Price
import boto.mturk.notification
from boto.connection import AWSQueryConnection
from boto.exception import EC2ResponseError
from boto.resultset import ResultSet
from boto.mturk.question import QuestionForm, ExternalQuestion, HTMLQuestion
class QualificationType(BaseAutoResultElement):
    """
    Class to extract an QualificationType structure from a response (used in
    ResultSet)

    Will have attributes named as per the Developer Guide,
    e.g. QualificationTypeId, CreationTime, Name, etc
    """
    pass
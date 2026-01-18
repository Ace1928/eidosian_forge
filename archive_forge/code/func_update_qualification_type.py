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
def update_qualification_type(self, qualification_type_id, description=None, status=None, retry_delay=None, test=None, answer_key=None, test_duration=None, auto_granted=None, auto_granted_value=None):
    params = {'QualificationTypeId': qualification_type_id}
    if description is not None:
        params['Description'] = description
    if status is not None:
        params['QualificationTypeStatus'] = status
    if retry_delay is not None:
        params['RetryDelayInSeconds'] = retry_delay
    if test is not None:
        assert isinstance(test, QuestionForm)
        params['Test'] = test.get_as_xml()
    if test_duration is not None:
        params['TestDurationInSeconds'] = test_duration
    if answer_key is not None:
        if isinstance(answer_key, basestring):
            params['AnswerKey'] = answer_key
        else:
            raise TypeError
    if auto_granted is not None:
        params['AutoGranted'] = auto_granted
    if auto_granted_value is not None:
        params['AutoGrantedValue'] = auto_granted_value
    return self._process_request('UpdateQualificationType', params, [('QualificationType', QualificationType)])
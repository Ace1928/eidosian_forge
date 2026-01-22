import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.support import exceptions
def add_communication_to_case(self, communication_body, case_id=None, cc_email_addresses=None, attachment_set_id=None):
    """
        Adds additional customer communication to an AWS Support case.
        You use the `CaseId` value to identify the case to add
        communication to. You can list a set of email addresses to
        copy on the communication using the `CcEmailAddresses` value.
        The `CommunicationBody` value contains the text of the
        communication.

        The response indicates the success or failure of the request.

        This operation implements a subset of the behavior on the AWS
        Support `Your Support Cases`_ web form.

        :type case_id: string
        :param case_id: The AWS Support case ID requested or returned in the
            call. The case ID is an alphanumeric string formatted as shown in
            this example: case- 12345678910-2013-c4c1d2bf33c5cf47

        :type communication_body: string
        :param communication_body: The body of an email communication to add to
            the support case.

        :type cc_email_addresses: list
        :param cc_email_addresses: The email addresses in the CC line of an
            email to be added to the support case.

        :type attachment_set_id: string
        :param attachment_set_id: The ID of a set of one or more attachments
            for the communication to add to the case. Create the set by calling
            AddAttachmentsToSet

        """
    params = {'communicationBody': communication_body}
    if case_id is not None:
        params['caseId'] = case_id
    if cc_email_addresses is not None:
        params['ccEmailAddresses'] = cc_email_addresses
    if attachment_set_id is not None:
        params['attachmentSetId'] = attachment_set_id
    return self.make_request(action='AddCommunicationToCase', body=json.dumps(params))
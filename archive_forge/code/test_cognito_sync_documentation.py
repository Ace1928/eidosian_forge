from boto.cognito.sync.exceptions import ResourceNotFoundException
from tests.integration.cognito import CognitoTest

    Even more so for Cognito Sync, Cognito identites are required.  However,
    AWS account IDs are required to aqcuire a Cognito identity so only
    Cognito pool identity related operations are tested.
    
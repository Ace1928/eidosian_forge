import os
from boto.compat import json
from boto.exception import JSONResponseError
from boto.connection import AWSAuthConnection
from boto.regioninfo import RegionInfo
from boto.awslambda import exceptions
def add_event_source(self, event_source, function_name, role, batch_size=None, parameters=None):
    """
        Identifies an Amazon Kinesis stream as the event source for an
        AWS Lambda function. AWS Lambda invokes the specified function
        when records are posted to the stream.

        This is the pull model, where AWS Lambda invokes the function.
        For more information, go to `AWS LambdaL How it Works`_ in the
        AWS Lambda Developer Guide.

        This association between an Amazon Kinesis stream and an AWS
        Lambda function is called the event source mapping. You
        provide the configuration information (for example, which
        stream to read from and which AWS Lambda function to invoke)
        for the event source mapping in the request body.

        This operation requires permission for the `iam:PassRole`
        action for the IAM role. It also requires permission for the
        `lambda:AddEventSource` action.

        :type event_source: string
        :param event_source: The Amazon Resource Name (ARN) of the Amazon
            Kinesis stream that is the event source. Any record added to this
            stream causes AWS Lambda to invoke your Lambda function. AWS Lambda
            POSTs the Amazon Kinesis event, containing records, to your Lambda
            function as JSON.

        :type function_name: string
        :param function_name: The Lambda function to invoke when AWS Lambda
            detects an event on the stream.

        :type role: string
        :param role: The ARN of the IAM role (invocation role) that AWS Lambda
            can assume to read from the stream and invoke the function.

        :type batch_size: integer
        :param batch_size: The largest number of records that AWS Lambda will
            give to your function in a single event. The default is 100
            records.

        :type parameters: map
        :param parameters: A map (key-value pairs) defining the configuration
            for AWS Lambda to use when reading the event source. Currently, AWS
            Lambda supports only the `InitialPositionInStream` key. The valid
            values are: "TRIM_HORIZON" and "LATEST". The default value is
            "TRIM_HORIZON". For more information, go to `ShardIteratorType`_ in
            the Amazon Kinesis Service API Reference.

        """
    uri = '/2014-11-13/event-source-mappings/'
    params = {'EventSource': event_source, 'FunctionName': function_name, 'Role': role}
    headers = {}
    query_params = {}
    if batch_size is not None:
        params['BatchSize'] = batch_size
    if parameters is not None:
        params['Parameters'] = parameters
    return self.make_request('POST', uri, expected_status=200, data=json.dumps(params), headers=headers, params=query_params)
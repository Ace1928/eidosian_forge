import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.datapipeline import exceptions
def activate_pipeline(self, pipeline_id):
    """
        Validates a pipeline and initiates processing. If the pipeline
        does not pass validation, activation fails.

        Call this action to start processing pipeline tasks of a
        pipeline you've created using the CreatePipeline and
        PutPipelineDefinition actions. A pipeline cannot be modified
        after it has been successfully activated.

        :type pipeline_id: string
        :param pipeline_id: The identifier of the pipeline to activate.

        """
    params = {'pipelineId': pipeline_id}
    return self.make_request(action='ActivatePipeline', body=json.dumps(params))
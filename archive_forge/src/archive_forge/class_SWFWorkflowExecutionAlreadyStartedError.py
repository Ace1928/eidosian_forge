from boto.exception import SWFResponseError
class SWFWorkflowExecutionAlreadyStartedError(SWFResponseError):
    """
    Raised when an open execution with the same workflow_id is already running
    in the specified domain.
    """
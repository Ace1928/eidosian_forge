from ray.util.annotations import PublicAPI
from ray.workflow.common import TaskID
@PublicAPI(stability='alpha')
class WorkflowCancellationError(WorkflowError):

    def __init__(self, workflow_id: str):
        self.message = f'Workflow[id={workflow_id}] is cancelled during execution.'
        super().__init__(self.message)
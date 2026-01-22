from parlai.core.worlds import World
from parlai.mturk.core.dev.agents import AssignState
class StaticMTurkTaskWorld(MTurkDataWorld):
    """
    World for handling generic tasks that aim to use ParlAI as an MTurk interface, but
    don't need the server to be in the loop.
    """

    def __init__(self, opt, mturk_agent, task_data):
        """
        Init should be provided with the task_data that the worker needs to complete the
        task on the frontend.
        """
        self.mturk_agent = mturk_agent
        self.episodeDone = False
        self.task_data = task_data

    def did_complete(self):
        """
        Determines whether or not this world was completed, or if the agent didn't
        complete the task.
        """
        agent = self.mturk_agent
        return not (agent.hit_is_abandoned or agent.hit_is_returned)

    def episode_done(self):
        """
        A ParlAI-MTurk task ends and allows workers to be marked complete when the world
        is finished.
        """
        return self.episodeDone

    def parley(self):
        """
        A static task parley is simply sending the task data and waiting for the
        response.
        """
        agent = self.mturk_agent
        agent.observe({'id': 'System', 'text': '[TASK_DATA]', 'task_data': self.task_data})
        agent.set_status(AssignState.STATUS_STATIC)
        self.response = agent.get_completed_act()
        self.episodeDone = True

    def prep_save_data(self, workers):
        """
        This prepares data to be saved for later review, including chats from individual
        worker perspectives.
        """
        custom_data = self.get_custom_task_data()
        save_data = {'custom_data': custom_data, 'worker_data': {}}
        agent = self.mturk_agent
        save_data['worker_data'][agent.worker_id] = {'worker_id': agent.worker_id, 'agent_id': agent.id, 'assignment_id': agent.assignment_id, 'task_data': self.task_data, 'response': self.response, 'completed': self.episode_done()}
        return save_data

    def shutdown(self):
        """
        Shutdown tracking for the agent.
        """
        self.mturk_agent.shutdown()
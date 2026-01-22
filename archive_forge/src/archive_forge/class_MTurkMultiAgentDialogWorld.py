from parlai.mturk.core.worlds import MTurkOnboardWorld, MTurkTaskWorld
from parlai.core.worlds import validate
from joblib import Parallel, delayed
class MTurkMultiAgentDialogWorld(MTurkTaskWorld):
    """
    Basic world where each agent gets a turn in a round-robin fashion, receiving as
    input the actions of all other agents since that agent last acted.
    """

    def __init__(self, opt, agents=None, shared=None):
        self.agents = agents
        self.acts = [None] * len(agents)
        self.episodeDone = False

    def parley(self):
        """
        For each agent, get an observation of the last action each of the other agents
        took.

        Then take an action yourself.
        """
        acts = self.acts
        for index, agent in enumerate(self.agents):
            try:
                acts[index] = agent.act(timeout=None)
            except TypeError:
                acts[index] = agent.act()
            if acts[index]['episode_done']:
                self.episodeDone = True
            for other_agent in self.agents:
                if other_agent != agent:
                    other_agent.observe(validate(acts[index]))

    def episode_done(self):
        return self.episodeDone

    def shutdown(self):
        """
        Shutdown all mturk agents in parallel, otherwise if one mturk agent is
        disconnected then it could prevent other mturk agents from completing.
        """
        global shutdown_agent

        def shutdown_agent(agent):
            try:
                agent.shutdown(timeout=None)
            except Exception:
                agent.shutdown()
        Parallel(n_jobs=len(self.agents), backend='threading')((delayed(shutdown_agent)(agent) for agent in self.agents))
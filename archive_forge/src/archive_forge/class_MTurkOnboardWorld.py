from parlai.core.worlds import World
from parlai.mturk.core.dev.agents import AssignState
class MTurkOnboardWorld(MTurkDataWorld):
    """
    Generic world for onboarding a Turker and collecting information from them.
    """

    def __init__(self, opt, mturk_agent):
        """
        Init should set up resources for running the onboarding world.
        """
        self.mturk_agent = mturk_agent
        self.episodeDone = False

    def parley(self):
        """
        A parley should represent one turn of your onboarding task.
        """
        self.episodeDone = True

    def episode_done(self):
        return self.episodeDone

    def review_work(self):
        """
        This call is an opportunity to act on this worker based on their onboarding
        work.

        Generally one could assign a qualification to soft block members who didn't pass
        the onboarding world.
        """
        pass

    def shutdown(self):
        """
        Clear up resources needed for this world.
        """
        pass
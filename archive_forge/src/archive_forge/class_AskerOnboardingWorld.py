from parlai.mturk.core.worlds import MTurkOnboardWorld, MTurkTaskWorld
import threading
class AskerOnboardingWorld(MTurkOnboardWorld):
    """
    Example onboarding world.

    Sends a message from the world to the worker and then exits as complete after the
    worker uses the interface
    """

    def parley(self):
        ad = {}
        ad['id'] = 'System'
        ad['text'] = "Welcome onboard! You'll be playing the role of the asker. Ask a question that can be answered with just a number. Send any message to continue."
        self.mturk_agent.observe(ad)
        self.mturk_agent.act()
        self.episodeDone = True
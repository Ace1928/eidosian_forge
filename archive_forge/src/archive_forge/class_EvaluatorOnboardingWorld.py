from parlai.mturk.core.worlds import MTurkOnboardWorld, MTurkTaskWorld
import threading
class EvaluatorOnboardingWorld(MTurkOnboardWorld):
    """
    Example onboarding world.

    Sends a message from the world to the worker and then exits as complete after the
    worker uses the interface
    """

    def parley(self):
        ad = {}
        ad['id'] = 'System'
        ad['text'] = "Welcome onboard! You'll be playing the evaluator. You'll observe a series of three questions, and then you'll evaluate whether or not the exchange was accurate. Send an eval to begin."
        self.mturk_agent.observe(ad)
        self.mturk_agent.act()
        self.episodeDone = True
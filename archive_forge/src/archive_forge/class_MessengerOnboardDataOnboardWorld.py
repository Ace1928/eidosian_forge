from parlai.core.worlds import World
from parlai.chat_service.services.messenger.worlds import OnboardWorld
class MessengerOnboardDataOnboardWorld(OnboardWorld):
    """
    Example messenger onboarding that collects and returns data for use in the real task
    world.
    """

    def __init__(self, opt, agent):
        self.agent = agent
        self.episodeDone = False
        self.turn = 0
        self.data = {}

    @staticmethod
    def generate_world(opt, agents):
        return MessengerOnboardDataOnboardWorld(opt=opt, agent=agents[0])

    @staticmethod
    def assign_roles(agents):
        for a in agents:
            a.disp_id = 'Agent'

    def parley(self):
        if self.turn == 0:
            self.agent.observe({'id': 'Onboarding', 'text': 'Welcome to the onboarding world the onboarding data demo.\nEnter your name.'})
            a = self.agent.act()
            while a is None:
                a = self.agent.act()
            self.data['name'] = a['text']
            self.turn = self.turn + 1
        elif self.turn == 1:
            self.agent.observe({'id': 'Onboarding', 'text': '\nEnter your favorite color.'})
            a = self.agent.act()
            while a is None:
                a = self.agent.act()
            self.data['color'] = a['text']
            self.episodeDone = True
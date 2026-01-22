from parlai.core.worlds import World
class OnboardWorld(World):

    def __init__(self, opt, agent):
        self.agent = agent
        self.episodeDone = False

    @staticmethod
    def generate_world(opt, agents):
        return OnboardWorld(opt, agents[0])

    def parley(self):
        self.episodeDone = True

    def episode_done(self):
        return self.episodeDone

    def shutdown(self):
        pass
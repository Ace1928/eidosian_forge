from parlai.core.worlds import World
class SimpleMessengerOverworld(World):
    """
    Passthrough world to spawn task worlds of only one type.

    Demos of more advanced overworld functionality exist in the overworld demo
    """

    def __init__(self, opt, agent):
        self.agent = agent
        self.opt = opt
        self.episodeDone = False

    def episode_done(self):
        return self.episodeDone

    @staticmethod
    def generate_world(opt, agents):
        return SimpleMessengerOverworld(opt, agents[0])

    def parley(self):
        self.episodeDone = True
        return 'default'
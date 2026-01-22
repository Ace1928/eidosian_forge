from autokeras.tuners import greedy
class TextClassifierTuner(greedy.Greedy):

    def __init__(self, **kwargs):
        super().__init__(initial_hps=TEXT_CLASSIFIER, **kwargs)
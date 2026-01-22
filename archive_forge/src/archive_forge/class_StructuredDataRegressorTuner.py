from autokeras.tuners import greedy
class StructuredDataRegressorTuner(greedy.Greedy):

    def __init__(self, **kwargs):
        super().__init__(initial_hps=STRUCTURED_DATA_REGRESSOR, **kwargs)
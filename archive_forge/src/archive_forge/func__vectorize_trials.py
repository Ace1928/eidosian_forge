import numpy as np
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.engine import hyperparameters as hp_module
from keras_tuner.src.engine import oracle as oracle_module
from keras_tuner.src.engine import trial as trial_module
from keras_tuner.src.engine import tuner as tuner_module
def _vectorize_trials(self):
    x = []
    y = []
    ongoing_trials = set(self.ongoing_trials.values())
    for trial in self.trials.values():
        trial_hps = trial.hyperparameters
        vector = []
        for hp in self._nonfixed_space():
            if trial_hps.is_active(hp) and hp.name in trial_hps.values:
                trial_value = trial_hps.values[hp.name]
            else:
                trial_value = hp.default
            prob = hp.value_to_prob(trial_value)
            vector.append(prob)
        if trial in ongoing_trials:
            x_h = np.array(vector).reshape((1, -1))
            y_h_mean, y_h_std = self.gpr.predict(x_h, return_std=True)
            y_h_mean = np.array(y_h_mean).flatten()
            score = y_h_mean[0] + y_h_std[0]
        elif trial.status == 'COMPLETED':
            score = trial.score
            if self.objective.direction == 'max':
                score = -1 * score
        elif trial.status in ['FAILED', 'INVALID']:
            continue
        x.append(vector)
        y.append(score)
    x = np.array(x)
    y = np.array(y)
    return (x, y)
import copy
import keras_tuner
import autokeras as ak
from autokeras.tuners import task_specific
def test_sd_clf_init_hp0_equals_hp_of_a_model(tmp_path):
    clf = ak.StructuredDataClassifier(directory=tmp_path, column_names=['a', 'b'], column_types={'a': 'numerical', 'b': 'numerical'})
    clf.inputs[0].shape = (2,)
    clf.outputs[0].in_blocks[0].shape = (10,)
    init_hp = task_specific.STRUCTURED_DATA_CLASSIFIER[0]
    hp = keras_tuner.HyperParameters()
    hp.values = copy.copy(init_hp)
    clf.tuner.hypermodel.build(hp)
    assert set(init_hp.keys()) == set(hp._hps.keys())
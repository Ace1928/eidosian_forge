import copy
import tqdm
from parlai.core.params import ParlaiParser
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task
import parlai.utils.logging as logging
from parlai.core.script import ParlaiScript, register_script
@register_script('extract_image_feature', hidden=True)
class ExtractImgFeatures(ParlaiScript):

    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        return extract_feats(self.opt)
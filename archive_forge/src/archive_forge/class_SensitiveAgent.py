from parlai.agents.transformer.transformer import TransformerClassifierAgent
from parlai.core.build_data import modelzoo_path
from parlai.utils.safety import OffensiveStringMatcher
from parlai.core.agents import add_datapath_and_model_args, create_agent_from_opt_file, create_agent
from openchat.base import ParlaiClassificationAgent, EncoderLM, SingleTurn
class SensitiveAgent(ParlaiClassificationAgent, EncoderLM, SingleTurn):

    def __init__(self, model, device, maxlen):
        option = self.set_options(name='zoo:sensitive_topics_classifier/model', device=device)
        super(SensitiveAgent, self).__init__(device=device, maxlen=maxlen, model=create_agent_from_opt_file(option), suffix='', name=model)

    def labels(self):
        return ['safe', 'politics', 'religion', 'medical_advice', 'dating', 'drugs']

    @staticmethod
    def available_models():
        return ['safety.sensitive']

    @staticmethod
    def default_maxlen():
        return 128

    def set_options(self, name, device):
        option = {}
        add_datapath_and_model_args(option)
        datapath = option.get('datapath')
        option['model_file'] = modelzoo_path(datapath, name)
        option = {'no_cuda': False if 'cuda' in device else True}
        if 'cuda:' in device:
            option['override']['gpu'] = int(device.split(':')[1])
        elif 'cuda' in device:
            option['override']['gpu'] = 0
        return option
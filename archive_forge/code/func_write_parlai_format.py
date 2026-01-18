from parlai.core.worlds import BatchWorld, DynamicBatchWorld
from parlai.utils.misc import msg_to_str
from parlai.utils.conversations import Conversations
import parlai.utils.logging as logging
import copy
from tqdm import tqdm
def write_parlai_format(self, outfile):
    logging.info(f'Saving log to {outfile} in ParlAI format')
    with open(outfile, 'w') as fw:
        for episode in tqdm(self._logs):
            ep = self.convert_to_labeled_data(episode)
            for act in ep:
                txt = msg_to_str(act)
                fw.write(txt + '\n')
            fw.write('\n')
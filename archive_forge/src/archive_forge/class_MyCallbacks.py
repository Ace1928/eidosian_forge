from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.utils.annotations import override
import numpy as np
class MyCallbacks(DefaultCallbacks):

    @override(DefaultCallbacks)
    def on_postprocess_trajectory(self, *, worker, episode, agent_id, policy_id, policies, postprocessed_batch, original_batches, **kwargs):
        super().on_postprocess_trajectory(worker=worker, episode=episode, agent_id=agent_id, policy_id=policy_id, policies=policies, postprocessed_batch=postprocessed_batch, original_batches=original_batches, **kwargs)
        if policies[policy_id].config.get('use_adapted_gae', False):
            policy = policies[policy_id]
            assert policy.config['use_gae'], "Can't use adapted gae without use_gae=True!"
            info_dicts = postprocessed_batch[SampleBatch.INFOS]
            assert np.all(['d_ts' in info_dict for info_dict in info_dicts]), "Info dicts in sample batch must contain data 'd_ts'                 (=ts[i+1]-ts[i] length of time steps)!"
            d_ts = np.array([np.float(info_dict.get('d_ts')) for info_dict in info_dicts])
            assert np.all([e.is_integer() for e in d_ts]), "Elements of 'd_ts' (length of time steps) must be integer!"
            if postprocessed_batch[SampleBatch.TERMINATEDS][-1]:
                last_r = 0.0
            else:
                input_dict = postprocessed_batch.get_single_step_input_dict(policy.model.view_requirements, index='last')
                last_r = policy._value(**input_dict)
            gamma = policy.config['gamma']
            lambda_ = policy.config['lambda']
            vpred_t = np.concatenate([postprocessed_batch[SampleBatch.VF_PREDS], np.array([last_r])])
            delta_t = postprocessed_batch[SampleBatch.REWARDS] + gamma ** d_ts * vpred_t[1:] - vpred_t[:-1]
            postprocessed_batch[Postprocessing.ADVANTAGES] = generalized_discount_cumsum(delta_t, d_ts[:-1], gamma * lambda_)
            postprocessed_batch[Postprocessing.VALUE_TARGETS] = (postprocessed_batch[Postprocessing.ADVANTAGES] + postprocessed_batch[SampleBatch.VF_PREDS]).astype(np.float32)
            postprocessed_batch[Postprocessing.ADVANTAGES] = postprocessed_batch[Postprocessing.ADVANTAGES].astype(np.float32)
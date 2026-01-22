import collections
import utils
import tensorflow as tf
def _unroll(self, prev_actions, env_outputs, core_state):
    unused_reward, done, unused_observation, _, _ = env_outputs
    torso_outputs = utils.batch_apply(self._torso, (prev_actions, env_outputs))
    initial_core_state = self._core.get_initial_state(batch_size=tf.shape(prev_actions)[1], dtype=tf.float32)
    core_output_list = []
    for input_, d in zip(tf.unstack(torso_outputs), tf.unstack(done)):
        core_state = tf.nest.map_structure(lambda x, y, d=d: tf.where(tf.reshape(d, [d.shape[0]] + [1] * (x.shape.rank - 1)), x, y), initial_core_state, core_state)
        core_output, core_state = self._core(input_, core_state)
        core_output_list.append(core_output)
    core_outputs = tf.stack(core_output_list)
    return (utils.batch_apply(self._head, (core_outputs,)), core_state)
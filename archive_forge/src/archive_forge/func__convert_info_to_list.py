from typing import List
import gym
def _convert_info_to_list(self, infos: dict) -> List[dict]:
    """Convert the dict info to list.

        Convert the dict info of the vectorized environment
        into a list of dictionaries where the i-th dictionary
        has the info of the i-th environment.

        Args:
            infos (dict): info dict coming from the env.

        Returns:
            list_info (list): converted info.

        """
    list_info = [{} for _ in range(self.num_envs)]
    list_info = self._process_episode_statistics(infos, list_info)
    for k in infos:
        if k.startswith('_'):
            continue
        for i, has_info in enumerate(infos[f'_{k}']):
            if has_info:
                list_info[i][k] = infos[k][i]
    return list_info
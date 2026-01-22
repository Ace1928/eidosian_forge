from typing import List
import gym
def _process_episode_statistics(self, infos: dict, list_info: list) -> List[dict]:
    """Process episode statistics.

        `RecordEpisodeStatistics` wrapper add extra
        information to the info. This information are in
        the form of a dict of dict. This method process these
        information and add them to the info.
        `RecordEpisodeStatistics` info contains the keys
        "r", "l", "t" which represents "cumulative reward",
        "episode length", "elapsed time since instantiation of wrapper".

        Args:
            infos (dict): infos coming from `RecordEpisodeStatistics`.
            list_info (list): info of the current vectorized environment.

        Returns:
            list_info (list): updated info.

        """
    episode_statistics = infos.pop('episode', False)
    if not episode_statistics:
        return list_info
    episode_statistics_mask = infos.pop('_episode')
    for i, has_info in enumerate(episode_statistics_mask):
        if has_info:
            list_info[i]['episode'] = {}
            list_info[i]['episode']['r'] = episode_statistics['r'][i]
            list_info[i]['episode']['l'] = episode_statistics['l'][i]
            list_info[i]['episode']['t'] = episode_statistics['t'][i]
    return list_info
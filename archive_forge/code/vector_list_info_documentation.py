from typing import List
import gym
Process episode statistics.

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

        
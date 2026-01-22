import argparse
import os
import re
import numpy as np
import ray
from ray import air, tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.utils import try_import_pyspiel, try_import_open_spiel
from ray.rllib.env.wrappers.open_spiel import OpenSpielEnv
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.examples.self_play_with_open_spiel import ask_user_for_action
from ray.rllib.policy.policy import PolicySpec
from ray.tune import register_env
from open_spiel.python.rl_environment import Environment  # noqa: E402
class LeagueBasedSelfPlayCallback(DefaultCallbacks):

    def __init__(self):
        super().__init__()
        self.main_policies = {'main', 'main_0'}
        self.main_exploiters = {'main_exploiter_0', 'main_exploiter_1'}
        self.league_exploiters = {'league_exploiter_0', 'league_exploiter_1'}
        self.trainable_policies = {'main', 'main_exploiter_1', 'league_exploiter_1'}
        self.non_trainable_policies = {'main_0', 'league_exploiter_0', 'main_exploiter_0'}
        self.win_rates = {}

    def on_train_result(self, *, algorithm, result, **kwargs):
        for policy_id, rew in result['hist_stats'].items():
            mo = re.match('^policy_(.+)_reward$', policy_id)
            if mo is None:
                continue
            policy_id = mo.group(1)
            won = 0
            for r in rew:
                if r > 0.0:
                    won += 1
            win_rate = won / len(rew)
            self.win_rates[policy_id] = win_rate
            if policy_id in self.non_trainable_policies:
                continue
            print(f"Iter={algorithm.iteration} {policy_id}'s win-rate={win_rate} -> ", end='')
            if win_rate > args.win_rate_threshold:
                is_main = re.match('^main(_\\d+)?$', policy_id)
                initializing_exploiters = False
                if is_main and len(self.trainable_policies) == 3:
                    initializing_exploiters = True
                    self.trainable_policies.add('league_exploiter_1')
                    self.trainable_policies.add('main_exploiter_1')
                else:
                    keep_training = False if is_main else np.random.choice([True, False], p=[0.3, 0.7])
                    if policy_id in self.main_policies:
                        new_pol_id = re.sub('_\\d+$', f'_{len(self.main_policies) - 1}', policy_id)
                        self.main_policies.add(new_pol_id)
                    elif policy_id in self.main_exploiters:
                        new_pol_id = re.sub('_\\d+$', f'_{len(self.main_exploiters)}', policy_id)
                        self.main_exploiters.add(new_pol_id)
                    else:
                        new_pol_id = re.sub('_\\d+$', f'_{len(self.league_exploiters)}', policy_id)
                        self.league_exploiters.add(new_pol_id)
                    if keep_training:
                        self.trainable_policies.add(new_pol_id)
                    else:
                        self.non_trainable_policies.add(new_pol_id)
                    print(f'adding new opponents to the mix ({new_pol_id}).')

                def policy_mapping_fn(agent_id, episode, worker, **kwargs):
                    type_ = np.random.choice([1, 2])
                    if type_ == 1:
                        league_exploiter = 'league_exploiter_' + str(np.random.choice(list(range(len(self.league_exploiters)))))
                        if league_exploiter not in self.trainable_policies:
                            opponent = np.random.choice(list(self.trainable_policies))
                        else:
                            opponent = np.random.choice(list(self.non_trainable_policies))
                        print(f'{league_exploiter} vs {opponent}')
                        return league_exploiter if episode.episode_id % 2 == agent_id else opponent
                    else:
                        main_exploiter = 'main_exploiter_' + str(np.random.choice(list(range(len(self.main_exploiters)))))
                        if main_exploiter not in self.trainable_policies:
                            main = 'main'
                        else:
                            main = np.random.choice(list(self.main_policies - {'main'}))
                        return main_exploiter if episode.episode_id % 2 == agent_id else main
                if initializing_exploiters:
                    main_state = algorithm.get_policy('main').get_state()
                    pol_map = algorithm.workers.local_worker().policy_map
                    pol_map['main_0'].set_state(main_state)
                    pol_map['league_exploiter_1'].set_state(main_state)
                    pol_map['main_exploiter_1'].set_state(main_state)
                    algorithm.workers.sync_weights(policies=['main_0', 'league_exploiter_1', 'main_exploiter_1'])

                    def _set(worker):
                        worker.set_policy_mapping_fn(policy_mapping_fn)
                        worker.set_is_policy_to_train(self.trainable_policies)
                    algorithm.workers.foreach_worker(_set)
                else:
                    new_policy = algorithm.add_policy(policy_id=new_pol_id, policy_cls=type(algorithm.get_policy(policy_id)), policy_mapping_fn=policy_mapping_fn, policies_to_train=self.trainable_policies)
                    main_state = algorithm.get_policy(policy_id).get_state()
                    new_policy.set_state(main_state)
                    algorithm.workers.sync_weights(policies=[new_pol_id])
                self._print_league()
            else:
                print('not good enough; will keep learning ...')

    def _print_league(self):
        print('--- League ---')
        print('Trainable policies (win-rates):')
        for p in sorted(self.trainable_policies):
            wr = self.win_rates[p] if p in self.win_rates else 0.0
            print(f'\t{p}: {wr}')
        print('Frozen policies:')
        for p in sorted(self.non_trainable_policies):
            wr = self.win_rates[p] if p in self.win_rates else 0.0
            print(f'\t{p}: {wr}')
        print()
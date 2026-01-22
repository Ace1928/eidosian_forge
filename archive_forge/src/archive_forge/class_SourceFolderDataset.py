import argparse
import random
from pathlib import Path
from typing import Optional, Union, Tuple, List, Any, Callable
import torch
import torch.utils.data
import torchaudio
import tqdm
class SourceFolderDataset(UnmixDataset):

    def __init__(self, root: str, split: str='train', target_dir: str='vocals', interferer_dirs: List[str]=['bass', 'drums'], ext: str='.wav', nb_samples: int=1000, seq_duration: Optional[float]=None, random_chunks: bool=True, sample_rate: float=44100.0, source_augmentations: Optional[Callable]=lambda audio: audio, seed: int=42) -> None:
        """A dataset that assumes folders of sources,
        instead of track folders. This is a common
        format for speech and environmental sound datasets
        such das DCASE. For each source a variable number of
        tracks/sounds is available, therefore the dataset
        is unaligned by design.
        By default, for each sample, sources from random track are drawn
        to assemble the mixture.

        Example
        =======
        train/vocals/track11.wav -----------------        train/drums/track202.wav  (interferer1) ---+--> input
        train/bass/track007a.wav  (interferer2) --/

        train/vocals/track11.wav ---------------------> output

        """
        self.root = Path(root).expanduser()
        self.split = split
        self.sample_rate = sample_rate
        self.seq_duration = seq_duration
        self.ext = ext
        self.random_chunks = random_chunks
        self.source_augmentations = source_augmentations
        self.target_dir = target_dir
        self.interferer_dirs = interferer_dirs
        self.source_folders = self.interferer_dirs + [self.target_dir]
        self.source_tracks = self.get_tracks()
        self.nb_samples = nb_samples
        self.seed = seed
        random.seed(self.seed)

    def __getitem__(self, index):
        audio_sources = []
        for source in self.source_folders:
            if self.split == 'valid':
                random.seed(index)
            source_path = random.choice(self.source_tracks[source])
            duration = load_info(source_path)['duration']
            if self.random_chunks:
                start = random.uniform(0, duration - self.seq_duration)
            else:
                start = max(duration // 2 - self.seq_duration // 2, 0)
            audio, _ = load_audio(source_path, start=start, dur=self.seq_duration)
            audio = self.source_augmentations(audio)
            audio_sources.append(audio)
        stems = torch.stack(audio_sources)
        x = stems.sum(0)
        y = stems[-1]
        return (x, y)

    def __len__(self):
        return self.nb_samples

    def get_tracks(self):
        """Loads input and output tracks"""
        p = Path(self.root, self.split)
        source_tracks = {}
        for source_folder in tqdm.tqdm(self.source_folders):
            tracks = []
            source_path = p / source_folder
            for source_track_path in sorted(source_path.glob('*' + self.ext)):
                if self.seq_duration is not None:
                    info = load_info(source_track_path)
                    if info['duration'] > self.seq_duration:
                        tracks.append(source_track_path)
                else:
                    tracks.append(source_track_path)
            source_tracks[source_folder] = tracks
        return source_tracks
from dataclasses import dataclass
@dataclass
class samsum_dataset:
    dataset: str = 'samsum_dataset'
    train_split: str = 'train'
    test_split: str = 'validation'
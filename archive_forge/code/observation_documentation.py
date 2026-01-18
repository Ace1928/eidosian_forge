import gym
import numpy as np
import tensorflow as tf
Wrapper that encodes a frame as packed bits instead of booleans.

  8x less to be transferred across the wire (16 booleans stored as uint16
  instead of 16 uint8) and 8x less to be transferred from CPU to TPU (16
  booleans stored as uint32 instead of 16 bfloat16).

  
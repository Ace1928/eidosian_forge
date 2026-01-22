import numpy as np  # Assuming NumPy is used for efficient array manipulation
import random
import types
import importlib.util
import logging
import collections
from typing import Deque, Dict, Tuple, List
from typing import List, Tuple
from functools import wraps
import logging
class DeepLearningDecisionMaker:
    """
    A class that encapsulates the neural network and provides methods for training, predicting, and simulating moves.
    """

    @StandardDecorator()
    def __init__(self):
        """
        Initializes the DeepLearningDecisionMaker with a custom NeuralNetwork and empty training data.
        """
        self.model = NeuralNetwork(16, [64, 64], 4)
        self.training_data = []
        self.target_scores = []
        logging.debug('DeepLearningDecisionMaker initialized with an empty model and training data.')

    @StandardDecorator()
    def train_network(self, training_data: np.ndarray, target_scores: np.ndarray, epochs: int=1000, learning_rate: float=0.01):
        """
        Trains the neural network model using the provided training data and target scores over a specified number of epochs.

        Args:
            training_data (np.ndarray): The input data for training.
            target_scores (np.ndarray): The target scores for each input.
            epochs (int): The number of training epochs.
            learning_rate (float): The learning rate for the gradient descent.
        """
        for epoch in range(epochs):
            predictions = self.model.predict(training_data)
            loss = np.mean((predictions - target_scores) ** 2)
            logging.info(f'Epoch {epoch + 1}/{epochs}, Loss: {loss}')

    @StandardDecorator()
    def backpropagation(self, input_data: np.ndarray, target: np.ndarray, learning_rate: float):
        """
        Performs a simplified backpropagation algorithm to adjust the weights and biases of the neural network.

        Args:
            input_data (np.ndarray): The input data used for training.
            target (np.ndarray): The target output.
            learning_rate (float): The learning rate for adjustments.
        """
        activations = [input_data]
        x = input_data
        for w, b in zip(self.model.weights[:-1], self.model.biases[:-1]):
            x = self.model.relu(np.dot(x, w) + b)
            activations.append(x)
        output = self.model.softmax(np.dot(x, self.model.weights[-1]) + self.model.biases[-1])
        activations.append(output)
        error = output - target
        for i in reversed(range(len(self.model.weights))):
            activation = activations[i]
            if i == len(self.model.weights) - 1:
                delta = error
            else:
                delta = np.dot(delta, self.model.weights[i + 1].T) * (activation > 0).astype(float)
            weight_gradient = np.dot(activations[i - 1].T, delta)
            bias_gradient = np.sum(delta, axis=0, keepdims=True)
            self.model.weights[i] -= learning_rate * weight_gradient
            self.model.biases[i] -= learning_rate * bias_gradient

    @StandardDecorator()
    def update_training_data(self, board_state: np.ndarray, score: int):
        """
        Updates the training data with the given board state and score.

        Args:
            board_state (np.ndarray): The current game board state.
            score (int): The score associated with the board state.
        """
        self.training_data.append(board_state.flatten())
        self.target_scores.append(score)

    @StandardDecorator()
    def train_model(self):
        """
        Trains the model using the collected training data if there is enough data collected.
        """
        if len(self.training_data) > 100:
            training_data_np = np.array(self.training_data)
            target_scores_np = np.array(self.target_scores)
            self.train_network(training_data_np, target_scores_np)

    @StandardDecorator()
    def predict_score(self, board_state: np.ndarray) -> float:
        """
        Predicts the score for a given board state using the neural network model.

        Args:
            board_state (np.ndarray): The current game board state.

        Returns:
            float: The predicted score.
        """
        prediction = self.model.predict([board_state.flatten()])[0]
        return prediction

    @StandardDecorator()
    def predict_move(self, board_state: np.ndarray) -> int:
        """
        Predicts the best move for a given board state using the neural network model.

        Args:
            board_state (np.ndarray): The current game board state.

        Returns:
            int: The index of the highest probability move.
        """
        prediction = self.model.predict(board_state.flatten().reshape(1, -1))
        return np.argmax(prediction)

    @StandardDecorator()
    def simulate_move(board: np.ndarray, move: str) -> Tuple[np.ndarray, int]:
        """
        Simulates a move on the board and returns the new board state and score gained.

        This function shifts the tiles in the specified direction and combines tiles of the same value.

        Args:
            board (np.ndarray): The current game board.
            move (str): The move to simulate ('up', 'down', 'left', 'right').

        Returns:
            Tuple[np.ndarray, int]: The new board state and score gained from the move.
        """

        @StandardDecorator()
        def shift_and_combine(row: list) -> Tuple[list, int]:
            """
            Shifts non-zero elements to the left and combines elements of the same value.
            Args:
                row (list): A row (or column) from the game board.
            Returns:
                Tuple[list, int]: The shifted and combined row, and the score gained.
            """
            non_zero = [i for i in row if i != 0]
            combined = []
            score = 0
            skip = False
            for i in range(len(non_zero)):
                if skip:
                    skip = False
                    continue
                if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                    combined.append(2 * non_zero[i])
                    score += 2 * non_zero[i]
                    skip = True
                else:
                    combined.append(non_zero[i])
            combined.extend([0] * (len(row) - len(combined)))
            return (combined, score)

        @StandardDecorator()
        def rotate_board(board: np.ndarray, move: str) -> np.ndarray:
            """
            Rotates the board to simplify shifting logic.
            Args:
                board (np.ndarray): The game board.
                move (str): The move direction.
            Returns:
                np.ndarray: The rotated board.
            """
            if move == 'up':
                return board.T
            elif move == 'down':
                return np.rot90(board, 2).T
            elif move == 'left':
                return board
            elif move == 'right':
                return np.rot90(board, 2)
            else:
                raise ValueError('Invalid move direction')
        rotated_board = rotate_board(board, move)
        new_board = np.zeros_like(board)
        total_score = 0
        for i, row in enumerate(rotated_board):
            new_row, score = shift_and_combine(list(row))
            total_score += score
            new_board[i] = new_row
        if move in ['up', 'down']:
            new_board = new_board.T
        elif move == 'right':
            new_board = np.rot90(new_board, 2)
        return (new_board, total_score)
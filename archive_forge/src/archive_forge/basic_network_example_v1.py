import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List

def init_params(layer_dims: List[int], initialization: str = 'he') -> Dict[str, np.ndarray]:
    """
    Initialize the parameters for a multi-layer neural network.
    """
    np.random.seed(3)
    params = {}
    L = len(layer_dims)

    for l in range(1, L):
        if initialization == 'he':
            params[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2. / layer_dims[l-1])
        elif initialization == 'xavier':
            params[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(1. / layer_dims[l-1])
        else:
            params[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        params[f'b{l}'] = np.zeros((layer_dims[l], 1))

    return params

def sigmoid(Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implements the sigmoid activation in a safe manner, avoiding overflow.
    """
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache

def relu(Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implements the ReLU activation function.
    """
    A = np.maximum(0, Z)
    cache = Z
    return A, cache

def relu_backward(dA: np.ndarray, cache: np.ndarray) -> np.ndarray:
    """
    Implements the backward propagation for a single ReLU unit.
    """
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def sigmoid_backward(dA: np.ndarray, cache: np.ndarray) -> np.ndarray:
    """
    Implements the backward propagation for a single sigmoid unit.
    """
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1-s)
    return dZ

def linear_forward(A: np.ndarray, W: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, tuple]:
    """
    Implement the linear part of a layer's forward propagation.
    """
    Z = W.dot(A) + b
    cache = (A, W, b)
    return Z, cache

def linear_activation_forward(A_prev: np.ndarray, W: np.ndarray, b: np.ndarray, activation: str) -> Tuple[np.ndarray, tuple]:
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer.
    """
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)
    return A, cache

def compute_cost(AL: np.ndarray, Y: np.ndarray, parameters: Dict[str, np.ndarray], lambd: float = 0) -> float:
    """
    Implement the cost function with L2 regularization.
    """
    m = Y.shape[1]
    cross_entropy_cost = -np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL)) / m

    L2_regularization_cost = 0
    if lambd > 0:
        L = len(parameters) // 2
        for l in range(1, L + 1):
            L2_regularization_cost += np.sum(np.square(parameters[f'W{l}']))
        L2_regularization_cost = L2_regularization_cost * lambd / (2 * m)

    cost = cross_entropy_cost + L2_regularization_cost
    return cost

def linear_backward(dZ: np.ndarray, cache: Tuple[np.ndarray, np.ndarray, np.ndarray], lambd: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Implement the linear portion of backward propagation for a single layer (layer l).
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T) / m + (lambd * W) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

def linear_activation_backward(dA: np.ndarray, cache: Tuple[tuple, np.ndarray], activation: str, lambd: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    """
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd)

    return dA_prev, dW, db

def update_parameters_with_momentum(parameters: Dict[str, np.ndarray], grads: Dict[str, np.ndarray], v: Dict[str, np.ndarray], beta: float, learning_rate: float) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Update parameters using gradient descent with momentum.
    """
    L = len(parameters) // 2  # number of layers in the neural network

    for l in range(L):
        v["dW" + str(l+1)] = beta * v["dW" + str(l+1)] + (1 - beta) * grads["dW" + str(l+1)]
        v["db" + str(l+1)] = beta * v["db" + str(l+1)] + (1 - beta) * grads["db" + str(l+1)]

        parameters["W" + str(l+1)] -= learning_rate * v["dW" + str(l+1)]
        parameters["b" + str(l+1)] -= learning_rate * v["db" + str(l+1)]

    return parameters, v

def initialize_velocity(parameters: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Initializes the velocity for momentum optimization.
    """
    L = len(parameters) // 2 # number of layers in the neural networks
    v = {}

    for l in range(L):
        v["dW" + str(l+1)] = np.zeros_like(parameters["W" + str(l+1)])
        v["db" + str(l+1)] = np.zeros_like(parameters["b" + str(l+1)])

    return v

def forward_propagation(X: np.ndarray, parameters: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[Tuple[np.ndarray]]]:
    """
    Implements forward propagation for the model: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
    """
    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network
    
    # Implement [LINEAR -> RELU]*(L-1)
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters[f'W{l}'], parameters[f'b{l}'], activation="relu")
        caches.append(cache)
    
    # Implement LINEAR -> SIGMOID
    AL, cache = linear_activation_forward(A, parameters[f'W{L}'], parameters[f'b{L}'], activation="sigmoid")
    caches.append(cache)
    
    return AL, caches

def backward_propagation(AL: np.ndarray, Y: np.ndarray, caches: List[Tuple[np.ndarray]], lambd: float) -> Dict[str, np.ndarray]:
    """
    Implements the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    """
    grads = {}
    L = len(caches) # the number of layers
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    # Lth layer (SIGMOID -> LINEAR) gradients
    current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation="sigmoid", lambd=lambd)
    
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation="relu", lambd=lambd)
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def model(X: np.ndarray, Y: np.ndarray, layers_dims: List[int], initialization: str = 'he', learning_rate: float = 0.0075, num_iterations: int = 3000, print_cost: bool = True, lambd: float = 0, beta: float = 0.9) -> Dict[str, np.ndarray]:
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    """
    np.random.seed(1)
    costs = []  # keep track of cost
    parameters = init_params(layers_dims, initialization=initialization)
    v = initialize_velocity(parameters)

    # Loop (gradient descent)
    for i in range(num_iterations):
        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID
        AL, caches = forward_propagation(X, parameters)
        
        # Compute cost
        cost = compute_cost(AL, Y, parameters, lambd)
        
        # Backward propagation
        grads = backward_propagation(AL, Y, caches, lambd)
        
        # Update parameters
        parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
        
        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print(f"Cost after iteration {i}: {cost:.6f}")
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # Plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title(f"Learning rate = {learning_rate}")
    plt.show()

    return parameters

import sklearn.model_selection

def normalize_data(X: np.ndarray) -> np.ndarray:
    """
    Normalize the feature dataset.
    
    Args:
        X (np.ndarray): Input data to be normalized.
        
    Returns:
        np.ndarray: Normalized data.
    """
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True)
    X_normalized = (X - mean) / std
    return X_normalized

def split_dataset(X: np.ndarray, Y: np.ndarray, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the dataset into training and test sets.
    
    Args:
        X (np.ndarray): The input data.
        Y (np.ndarray): The labels.
        test_size (float): The proportion of the dataset to include in the test split.
        
    Returns:
        Tuple containing the training data, test data, training labels, and test labels.
    """
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X.T, Y.T, test_size=test_size, random_state=1)
    return X_train.T, X_test.T, Y_train.T, Y_test.T

def evaluate_model(AL: np.ndarray, Y: np.ndarray) -> Dict[str, float]:
    """
    Evaluate the model's performance using precision, recall, and F1 score.
    
    Args:
        AL (np.ndarray): Probability vector corresponding to label predictions, shape (1, number of examples).
        Y (np.ndarray): True "label" vector (containing 0 if non-cat, 1 if cat), shape (1, number of examples).
        
    Returns:
        Dict[str, float]: Dictionary containing precision, recall, and F1 score.
    """
    predictions = AL >= 0.5
    true_positives = np.sum(predictions * Y)
    predicted_positives = np.sum(predictions)
    actual_positives = np.sum(Y)
    
    precision = true_positives / predicted_positives if predicted_positives > 0 else 0
    recall = true_positives / actual_positives if actual_positives > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {"precision": precision, "recall": recall, "f1_score": f1_score}

def predict(X: np.ndarray, parameters: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Use the learned parameters to predict labels for a dataset X.
    
    Args:
        X (np.ndarray): Input data to predict.
        parameters (Dict[str, np.ndarray]): Parameters of the trained model.
        
    Returns:
        np.ndarray: Predictions (0/1) for the input dataset.
    """
    AL, _ = forward_propagation(X, parameters)
    predictions = AL >= 0.5
    return predictions

# Example usage of the utility functions
# Assume X and Y are already loaded into the environment

# Normalize the dataset
X_normalized = normalize_data(X)

# Split the dataset into training and test sets
X_train, X_test, Y_train, Y_test = split_dataset(X_normalized, Y)

# Train the model using the training set
layers_dims = [X_train.shape[0], 20, 7, 5, 1]  # Example layer dimensions
parameters = model(X_train, Y_train, layers_dims, initialization='he', num_iterations=2500, print_cost=True)

# Evaluate the model on the test set
AL_train, _ = forward_propagation(X_train, parameters)
AL_test, _ = forward_propagation(X_test, parameters)
train_evaluation = evaluate_model(AL_train, Y_train)
test_evaluation = evaluate_model(AL_test, Y_test)

print(f"Training Evaluation: {train_evaluation}")
print(f"Test Evaluation: {test_evaluation}")

# Predictions
predictions_train = predict(X_train, parameters)
predictions_test = predict(X_test, parameters)



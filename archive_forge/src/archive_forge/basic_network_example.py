import numpy as np

def init_params(layer_dims: list, initialization: str = 'he') -> dict:
    """
    Initialize the parameters for a multi-layer neural network.

    Args:
        layer_dims (list): Dimensions of each layer in the network.
        initialization (str): The method of initialization ('he' or 'random').

    Returns:
        dict: A dictionary containing the initialized weights and biases.
    """
    np.random.seed(3)
    params = {}
    L = len(layer_dims)

    for l in range(1, L):
        if initialization == 'he':
            params[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2. / layer_dims[l-1])
        else:
            params[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        params[f'b{l}'] = np.zeros((layer_dims[l], 1))

    return params

def sigmoid(Z: np.ndarray) -> tuple:
    """
    Implements the sigmoid activation in a safe manner, avoiding overflow.

    Args:
        Z (np.ndarray): The input value(s).

    Returns:
        tuple: The activation value(s) and the cache containing Z.
    """
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache

def relu(Z: np.ndarray) -> tuple:
    """
    Implements the ReLU activation function.

    Args:
        Z (np.ndarray): The input value(s).

    Returns:
        tuple: The activation value(s) and the cache containing Z.
    """
    A = np.maximum(0, Z)
    cache = Z
    return A, cache

def relu_backward(dA: np.ndarray, cache: np.ndarray) -> np.ndarray:
    """
    Implements the backward propagation for a single ReLU unit.

    Args:
        dA (np.ndarray): Post-activation gradient.
        cache (np.ndarray): 'Z' where we store for computing backward propagation efficiently.

    Returns:
        np.ndarray: Gradient of the cost with respect to Z.
    """
    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.
    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0
    return dZ

def sigmoid_backward(dA: np.ndarray, cache: np.ndarray) -> np.ndarray:
    """
    Implements the backward propagation for a single sigmoid unit.

    Args:
        dA (np.ndarray): Post-activation gradient.
        cache (np.ndarray): 'Z' where we store for computing backward propagation efficiently.

    Returns:
        np.ndarray: Gradient of the cost with respect to Z.
    """
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1-s)
    return dZ

def linear_forward(A: np.ndarray, W: np.ndarray, b: np.ndarray) -> tuple:
    """
    Implement the linear part of a layer's forward propagation.

    Args:
        A (np.ndarray): Activations from previous layer (or input data).
        W (np.ndarray): Weights matrix.
        b (np.ndarray): Bias vector.

    Returns:
        tuple: The linear cache and the linear hypothesis Z.
    """
    Z = W.dot(A) + b
    cache = (A, W, b)
    return Z, cache

def linear_activation_forward(A_prev: np.ndarray, W: np.ndarray, b: np.ndarray, activation: str) -> tuple:
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer.

    Args:
        A_prev (np.ndarray): activations from previous layer (or input data).
        W (np.ndarray): weights matrix.
        b (np.ndarray): bias vector.
        activation (str): the activation to be used in this layer, stored as a text string: "sigmoid" or "relu".

    Returns:
        tuple: The activation value from the current layer and the cache.
    """
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)
    return A, cache

def compute_cost(AL: np.ndarray, Y: np.ndarray, parameters: dict, lambd: float = 0) -> float:
    """
    Implement the cost function with L2 regularization.

    Args:
        AL (np.ndarray): Probability vector corresponding to your label predictions, shape (1, number of examples).
        Y (np.ndarray): True "label" vector.
        parameters (dict): A dictionary containing the parameters of the model.
        lambd (float): Regularization hyperparameter.

    Returns:
        float: The adjusted cost.
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

def linear_backward(dZ: np.ndarray, cache: tuple, lambd: float) -> tuple:
    """
    Implement the linear portion of backward propagation for a single layer (layer l).

    Args:
        dZ (np.ndarray): Gradient of the cost with respect to the linear output (of current layer l).
        cache (tuple): Tuple containing (A_prev, W, b) from the forward propagation in the current layer.
        lambd (float): Regularization hyperparameter.

    Returns:
        tuple: Gradients of the cost with respect to A_prev, W, and b respectively.
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T) / m + (lambd * W) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

def linear_activation_backward(dA: np.ndarray, cache: tuple, activation: str, lambd: float) -> tuple:
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    Args:
        dA (np.ndarray): Post-activation gradient for current layer l.
        cache (tuple): Tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently.
        activation (str): The activation to be used in this layer, stored as a text string: "sigmoid" or "relu".
        lambd (float): Regularization hyperparameter.

    Returns:
        tuple: Gradients of the cost with respect to A_prev, W, and b respectively.
    """
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd)

    return dA_prev, dW, db

def update_parameters_with_momentum(parameters: dict, grads: dict, v: dict, beta: float, learning_rate: float) -> tuple:
    """
    Update parameters using gradient descent with momentum.

    Args:
        parameters (dict): Dictionary containing your parameters.
        grads (dict): Dictionary containing your gradients for each parameters.
        v (dict): Momentum - moving average of the gradients.
        beta (float): The momentum hyperparameter.
        learning_rate (float): The learning rate.

    Returns:
        tuple: Updated parameters and v.
    """
    L = len(parameters) // 2  # number of layers in the neural network

    for l in range(L):
        v["dW" + str(l+1)] = beta * v["dW" + str(l+1)] + (1 - beta) * grads["dW" + str(l+1)]
        v["db" + str(l+1)] = beta * v["db" + str(l+1)] + (1 - beta) * grads["db" + str(l+1)]

        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * v["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * v["db" + str(l+1)]

    return parameters, v

def model(X: np.ndarray, Y: np.ndarray, layers_dims: list, learning_rate: float = 0.0075, num_iterations: int = 3000, print_cost: bool = True, lambd: float = 0, beta: float = 0.9) -> dict:
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    lambd -- regularization hyperparameter, scalar
    beta -- Momentum hyperparameter, scalar

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []                         # keep track of cost
    parameters = init_params(layers_dims)
    v = initialize_velocity(parameters)

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = forward_propagation(X, parameters)
        
        # Compute cost.
        cost = compute_cost(AL, Y, parameters, lambd)
        
        # Backward propagation.
        grads = backward_propagation(AL, Y, caches, lambd)
        
        # Update parameters.
        parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
        
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters

def initialize_velocity(parameters: dict) -> dict:
    """
    Initializes the velocity as a python dictionary with:
                - keys: "dW1", "db1", ..., "dWL", "dbL" 
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    Returns:
    v -- python dictionary containing the current velocity.
                    v['dW' + str(l)] = velocity of dWl
                    v['db' + str(l)] = velocity of dbl
    """

    L = len(parameters) // 2 # number of layers in the neural networks
    v = {}

    for l in range(L):
        v["dW" + str(l+1)] = np.zeros_like(parameters["W" + str(l+1)])
        v["db" + str(l+1)] = np.zeros_like(parameters["b" + str(l+1)])

    return v


